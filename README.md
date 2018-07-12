# Recurrent Predictive State Policy Networks

This is Python 2.7 code for the paper <a href='https://arxiv.org/pdf/1803.01489.pdf'>"Recurrent Predictive State Policy Networks"</a> published at ICML 2018. 

## Instalation

<p> Install the package: </p>
<pre><code>python setup install</code></pre>

<p> Make sure you have the package installed.
 Open a python terminal and import the package: </p>
<pre><code>import rpsp</code></pre>


## Testing

<p> To run the code, use the following command: </p>

<pre><code>python run/call_test.py [--config <parameter_file>] [--other_key_args]</code></pre>. 

<p> A parameter_file stores command line arguments. Sample parameter files are located in test folder. You can override some of the parameters by using the appropriate commandline arguments AFTER specifying the parameter_file. For example, this command uses the settings in tests/CartPole-v1/Alt+obs/params (alternating optimization) but uses CatrPole-v0 environment and stores results in 'results' folder:</p>
<pre><code>python run/call_test.py --config 'tests/CartPole-v1/Alt+obs/params' --env CartPole-v0 --tfile results</code></pre>

<p> Check available environments in (envs.load_environments). For example: Swimmer-v0 runs a continuous simulator while Swimmer-v1 runs with the Mujoco simulator.
 Results in the paper are reported on Mujoco simulators.</p>

* * *

<p> For questions please email:
<p> ahefny@cs.cmu.edu, zmarinho@cmu.edu</p>



