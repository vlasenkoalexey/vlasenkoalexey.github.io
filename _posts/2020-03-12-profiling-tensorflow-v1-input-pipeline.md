---
layout: post
title: "Profiling TensorFlow v1 input pipeline and analyzing its performance using BigQuery"
date: 2020-03-12
categories: [Machine Learning, TensorFlow]
tags: [BigQuery, ML, Profiling, TensorFlow]
original_url: "https://alekseyv-dev.blogspot.com/2020/03/profiling-tensorflow-v1-input-pipeline.html"
---

Recently I had to debug a performance issue in TensorFlow V1.x input pipeline. TensorFlow V2.x has a nice tensorboard profiler plugin which is super handy for such situations. Unfortunately it doesn't work for TensorFlow V1.x, and I wasn't able to find a lot of information on debugging performance in TensorFlow V1.x. So I'll share my findings, since a lot of people are still on TF1.x, and this information might be useful. Most of TF1.x code is built using Estimators API. For profiling Estimators there is a tf.estimator.ProfilerHook, which can save traces every N steps or N seconds. It is pretty easy to use:
Sample showing how to train models in TensorFlow 1.14/1.15 using estimator API

```
def train_estimator_linear(model_dir):
  global ARGS

  logging.info('training for {} steps'.format(get_max_steps()))
  config = tf.estimator.RunConfig().replace(save_summary_steps=10)

  hooks = []
  if ARGS.profiler:
    profiler_hook = tf.estimator.ProfilerHook(
    save_steps=get_training_steps_per_epoch(),
    output_dir=os.path.join(model_dir, "profiler"),
    show_dataflow=True,
    show_memory=True)
    hooks.append(profiler_hook)

  feature_columns = create_feature_columns()
  estimator = tf.estimator.LinearClassifier(
      feature_columns=feature_columns,
      optimizer=GradientDescentOptimizer(learning_rate=0.001),
      model_dir=model_dir,
      config=config
  )
  logging.info('training and evaluating linear estimator model')
  tf.estimator.train_and_evaluate(
      estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=lambda: get_dataset('train'), max_steps=get_max_steps(), hooks=hooks),
      eval_spec=tf.estimator.EvalSpec(input_fn=lambda: get_dataset('test')))
  logging.info('done evaluating estimator model')
```

Once you run this code, it'll save bunch of timeline\_xxx.json files under model\_dir/profiler.
You can open those files using chrome profiler viewer - navigate to chrome://tracing and open one of json files.
Results looks like this:

![Estimator profile](/assets/images/profiling-tensorflow-v1-input-pipeline/image-1.png)

This works, also there are copule of problems. It is only possible to see profile for one iteration. In some cases that is sufficient, but sometimes it is necessary to analyze multiple iterations to find an issue.
And this only works for Esitmator API.

Luckily TensorFlow has a lower level API to profile single operations.

```
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
sess.run(<values_you_want_to_execute>, options=run_options, run_metadata=run_metadata)
tl = timeline.Timeline(op_metadata)
ctf = tl.generate_chrome_trace_format()
with open(os.path.join(model_dir, 'timeline.json'), 'w') as f:
    f.write(ctf)
```

This also dumps profile for a single op, which in a lot of cases doesn't show the whole picture.
Chrome only allows loading one timeline.json file at a time.

There are couple of ways to address this.

- Manually combine data from multiple traces by manipulating JSON.
- Import JSON trace files into BigQuery for further analysys.

Let's see how to use both approaches.

## Combining traces from multiple ops

Internally traces look like this:

```
{
    "traceEvents": [
        // ....
        {
            "ph": "X",
            "cat": "Op",
            "name": "unknown",
            "pid": 1,
            "tid": 0,
            "ts": 1584128411446459,
            "dur": 4763,
            "args": {
                "name": "Iterator::Model::Map::Prefetch::ParseExample",
                "op": "unknown"
            }
        },
        {
            "ph": "X",
            "cat": "Op",
            "name": "unknown",
            "pid": 1,
            "tid": 1,
            "ts": 1584128411446994,
            "dur": 2,
            "args": {
                "name": "linear/linear_model/linear_model/linear_model/cat14/Shape/Cast:Cast",
                "op": "unknown"
            }
        },
        // ....
    ]
}
```

Few things to notice:

- All events are grouped under "traceEvents"
- dur is duration of event in ms
- ts is some absolute timestamp

So in order to combinde few files it is necessary to load all files, extract content
under "traceEvents" and dump into one file under "traceEvents".

JSON manipulations in Python are pretty straightforward. Here is a sample:

```
 run_options = RunOptions(trace_level=RunOptions.FULL_TRACE)
 run_metadata= RunMetadata()
 ops_metadata = []

 with tf.compat.v1.Session() as sess:
   # some code here
   sess.run(<op to profile>, options=run_options, run_metadata=run_metadata)
   ops_metadata.append(run_metadata.step_stats)

 all_trace_events = []
 for op_metadata in ops_metadata:
     tl = timeline.Timeline(op_metadata)
     ctf = tl.generate_chrome_trace_format()
     trace_events = json.loads(ctf)['traceEvents']
     all_trace_events.extend(trace_events)

 with open(os.path.join(model_dir, 'timeline.json'), 'w') as f:
     f.write(json.dumps({'traceEvents' : all_trace_events}).replace('\n', ' '))
```

Note that geterated file might be too large, so you might want to limit number of iterations to profile.
Result will look similar to first example, but will contain information about multiple operations.

![Issue profile](/assets/images/profiling-tensorflow-v1-input-pipeline/image-2.png)

This trace clearly indicates that there is a MatchingFiles op takes too long.

Still it wasn't the root cause for the issue I was reserching. To get more in-depth stats to analyze
data I imported data into BigQuery.

## Importing JSON trace files into BigQuery

BigQuery is a data warehouse engine that can consume data in multiple formats (JSON is natively supported),
and allows to analyze that data using SQL.
BigQuery has no problems digesting hundred megabyte files in seconds.

Turnes out that importing JSON into BigQuery is very straingforward. It is not even necessary to specify
schema, BigQuery is clever enought to infer it on it's own. One catch is that each entry has to be
on a separate line. Here is a function to import data:

```
def import_json_file(dataset_id, table_id, file_name):
    client = bigquery.Client(project=PROJECT_ID)

    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    job_config.autodetect = True

    with open(file_name, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()  # Waits for table load to complete.

    print("Loaded {} rows into {}:{}.".format(job.output_rows, dataset_id, table_id))
```

Since timeline files have everything nested under "traceEvents", imported table is not
very conventient to work with. It would be nice to flatten it:

```
%%bigquery --project alekseyv-scalableai-dev --verbose --destination_table traces.tf115_processed

SELECT
index,
traceEvent.ph,
traceEvent.cat,
traceEvent.name,
traceEvent.pid,
traceEvent.tid,
traceEvent.ts,
traceEvent.dur,
traceEvent.args.name as arg_name,
traceEvent.args.op as arg_op
FROM
`alekseyv-scalableai-dev.traces.tf115` AS t,
UNNEST(t.traceEvents) as traceEvent WITH OFFSET AS index;
```

As a result of execution this query will create a new `alekseyv-scalableai-dev.traces.tf115`
table, see --destination\_table traces.tf115\_processed paramter.
Note that currenly --destination\_table is not yet available in Colab by default.
In order to enable it you'd need to update BigQuery pip package:

```
!pip install --upgrade google-cloud-bigquery
```

I really missed this feature in %%bigquery magic, so decided to add it myself recently.

Once data is in appropriate format, it is possible to analyze it using full power of SQL.
For example:

```
%%bigquery --project alekseyv-scalableai-dev tf115_processed

SELECT
sum(dur) as total_dur,
avg(dur) as avg_dur,
min(dur) as min_dur,
max(dur) as max_dur,
arg_name
FROM `alekseyv-scalableai-dev.traces.tf115_processed`
group by arg_name order by 1 desc
limit 20;
```

Full code used here and Jupyter notebook are available at: <https://github.com/vlasenkoalexey/columnar_estimator_sample>
