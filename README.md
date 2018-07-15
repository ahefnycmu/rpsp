# Recurrent Predictive State Policy Networks

This is Python 2.7 code for the paper <a href='https://arxiv.org/pdf/1803.01489.pdf'>"Recurrent Predictive State Policy Networks"</a> published at ICML 2018. 

## Instalation

<p> Install the package: </p>
<pre><code>python setup.py install</code></pre>

<p> Make sure you have the package installed.
 Open a python terminal and import the package: </p>
<pre><code>import rpsp</code></pre>


## Testing

<p> To run the code, use the following command: </p>

<pre><code>python rpsp/run/call_test.py [--config <parameter_file>] [--other_key_args]</code></pre>


<p> A parameter file stores command line arguments. Sample parameter files are located in test folder. 
You can override some of the parameters by using the appropriate commandline arguments AFTER specifying the parameter file. 
For example, this command uses the settings in tests/CartPole-v1/Alt+obs/params (alternating optimization) but runs for 500 iterations environment and stores results in 'results' folder:</p>
<pre><code>python rpsp/run/call_test.py --config 'tests/CartPole-v1/Alt+obs/params' --iter 500 --tfile results</code></pre>

<p>For additional parameter options please refer to the file "rpsp/run/call_test.py".</p>
* * *

<p> For questions please email:
<p> ahefny@cs.cmu.edu, zmarinho@cmu.edu</p>



