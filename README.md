# Recurrent Predictive State Policy Networks



## Instalation

<p> Install the package: </p>
<pre><code>python setup install</code></pre>

<p> Make sure you have the package installed.
 Open a python terminal and import the package: </p>
<pre><code>import rpsp</code></pre>


## Testing

<p> To Run pre-specified test files. You can tune any of the parameters in rpsp.run.call_test.</p>

<pre><code>python run/call_test.py --config <parameter_file> --other_key_args</code></pre>

<p> For example for CatrPole-v1 with alternating optimization and storing results in 'results' folder:</p>
<pre><code>python run/call_test.py --config 'tests/CartPole-v1/Alt+obs/params' --env CartPole-v0 --tfile results</code></pre>

<p> Check available environments in (envs.load_environments). For example: Swimmer-v0 runs a continuous simulator while Swimmer-v1 runs with the Mujoco simulator.
 Results in the paper are reported on different continuous Mujoco simulators.</p>


* * *

<p> Please refer to the paper for further details:
<p><a href='https://arxiv.org/pdf/1803.01489.pdf'> https://arxiv.org/pdf/1803.01489.pdf </a></p>


<p> For questions please email the authors:
<p> ahefny@cs.cmu.edu, zmarinho@cmu.edu</p>



