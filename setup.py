from setuptools import setup
from distutils.core import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='rpsp',
      version='0.1',
      description='Recurrent Predictive State Policy Networks',
      long_description=readme(),
      url='https://github.com/ahefnycmu/rpsp',
      author='Zita Marinho, Ahmed Hefny',
      author_email='zmarinho@cmu.edu, ahefny@cmu.edu',
      license='MIT',
      packages=['rpsp', 'rpsp.envs', 'rpsp.explore', 'rpsp.filters', 'rpsp.policy_opt', 'rpsp.rpspnets',
                'rpsp.rpspnets.psr_lite','rpsp.rpspnets.psr_lite.utils', 'rpsp.rpspnets._test', 'rpsp.run',
                'rpsp.run.test_utils', 'rpsp.policy'],
      install_requires=[
          'markdown',
      ],
      include_package_data=True,
      zip_safe=False)