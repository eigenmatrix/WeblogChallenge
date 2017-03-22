import os
from datetime import datetime
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD

os.environ["SPARK_HOME"] = "/Users/pacman/Documents/spark-2.1.0-bin-hadoop2.7/"
from pyspark import SparkContext


def convert_tp(tp):
    return (datetime.strptime(tp, '%Y-%m-%dT%H:%M:%S.%fZ') - datetime(1970, 1, 1)).total_seconds()


def clean_data(x):
    fields = x.split()
    return ((fields[2], convert_tp(fields[0])), (fields[1], fields[3], fields[12]))


def clean_data_for_load(x):
    fields = x.split()
    return (int(convert_tp(fields[0])/60), 1)


def clean_data_for_load2(x):
    fields = x.split()
    return (int(convert_tp(fields[0])), 1)

def sessionize(data):
    prev = None
    start = 0
    output = []
    session_out = set()
    user_ip, l = data
    size = len(l) - 1
    for c, (timestamp, metadata) in enumerate(l):
        if not prev:
            prev = timestamp
            start = timestamp
        if timestamp - prev <= 900:
            session_out.add(metadata)
        else:
            delta = prev - start
            start = timestamp
            output.append((delta, len(session_out)))
            session_out = set([metadata])
        if c == size:
            delta = timestamp - start
            session_out.add(metadata)
            output.append((delta, len(session_out)))
        prev = timestamp

    # output (user_ip, (session_time, num_uniq_url))
    return (user_ip, output)


def prepare_data_for_regression(data):
    user_ip, l = data
    if len(l) < 3: return
    for i in xrange(0, len(l) - 2):
        yield LabeledPoint(l[i + 2][1], [l[i][1], l[i + 1][1]])


def ave_session(kv):
    key, val = kv
    return (key, sum([v[0] for v in val]) / len(val))


def train_regression(data):
    model = LinearRegressionWithSGD.train(data, iterations=100, step=0.00000001)
    valuesAndPreds = data.map(lambda p: (p.label, model.predict(p.features)))
    MSE = valuesAndPreds \
              .map(lambda (v, p): (v - p) ** 2) \
              .reduce(lambda x, y: x + y) / valuesAndPreds.count()
    print("Mean Squared Error = " + str(MSE))
    return model


def prepare_data_for_test(data):
    user_ip, l = data
    if len(l) < 2:
        return (user_ip, LabeledPoint(0, [-10, l[-1][1]]))
    return (user_ip, LabeledPoint(0, [l[-2][1], l[-1][1]]))


if __name__ == "__main__":
    sc = SparkContext()
    lines = sc.textFile("/Users/pacman/git/paytm/data/2015_07_22_mktplace_shop_web_log_sample.log", 1)

    # Sessionized data with user ip
    sessionized = lines.map(clean_data) \
        .sortByKey().map(lambda x: (x[0][0], (x[0][1], x[1]))) \
        .groupByKey() \
        .map(sessionize)

    # Average session time per user ip
    ave_session_time = sessionized.map(ave_session)

    # Most engaged user
    engaged_user = sessionized.flatMap(lambda x: [(tp[0], x[0]) for tp in x[1]]) \
        .reduce(lambda x, y: x if x[0] > y[0] else y)

    # Do a two lag regression to predict session time and uniq number of url visited
    # if the user haven't have enough session yet, uses the last session time of url count as the prediction
    data = sessionized.flatMap(prepare_data_for_regression)
    model = train_regression(data)
    test = sessionized.map(prepare_data_for_test)
    pred = test.map(lambda p: (p[0], model.predict(p[1].features) if p[1].features[0] != -10 else p[1].features[1]))
    # the average number of session is less than 2 so the above model doesn't work,
    # the best guess would be the same as previous session
    ave_num_session = sessionized.map(lambda x: (1, len(x[1]))).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    # print ave_num_session

    # To use an heurestic if the timestamp in minutes ends with the number, then
    #  the load will be match to the average
    requests_by_minute = lines.map(clean_data_for_load).reduceByKey(lambda a, b: a + b)\
        .map(lambda x: (x[0]% 10, x[1])).groupByKey()\
        .map(lambda x: (x[0], sum(x[1])*1./len(x[1])))

    # The output is stored in the combined.txt file, and it looks like there is no better way to predicting the
    # requests, we can sessionize the data perhaps and trying to fit a quadratic curve to it. Alternatively,
    # using a moving average to approxi the prediction
    requests_by_secs = lines.map(clean_data_for_load2).reduceByKey(lambda a, b: a + b).sortByKey()
    sc.stop()
