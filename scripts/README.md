# Scripts

## agent_runner.py

A script to run "submitted" agents against one another and visualise the game. If only one agent file is provided, the agent is run against the kaggle-provided negamax agent.

Example usage:

```bash
python agent_runner.py agent_file_1.py [optional: agent_file_2.py]
```

Where `agent_file` has been created by `submit_agent.py`

## submit_agent.py

A script to submit an agent in a python file to be executable by `agent_runner.py`. From the Kaggle website:

> To create the submission, an agent function should be fully encapsulated (no external dependencies). When your agent is being evaluated against others, it will not have access to the Kaggle docker image. Only the following can be imported: Python Standard Library Modules, gym, numpy, scipy, pytorch (1.3.1, cpu only), and more may be added later.

These modules are in the common conda environment, so use this to check that your agent will run.

Example usage:

```bash
python submit_agent.py agent_file.py
```

Note: The "encapsulated function" should contain a function called `my_agent(observation, configuration)` where observation and configuration can be obtained from env.observation and env.configuration. Env should be created with `kaggle_environments.make("connectx")`

See `example_submission/example_agent.py` to see this script in use, and the expected, runnable output is in `example_submission/example_submission.py`

## End

