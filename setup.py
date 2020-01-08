from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym_city',
      version='0.0.0',
      install_requires=['gym', 
          'numpy', 
          'pillow', 
          'baselines', 
          'imutils', 
          'visdom',
          'graphviz'],
      author="Sam Earle",
      author_email="smearle93@gmail.com",
      description="An OpenAI Gym interface for Micropolis.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/smearle/gym-city",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
      ]
)


setup(name='MicropolisEnv-v0',
      version='0.0.1',
      install_requires=['gym']  # And any other dependencies foo needs
)

setup(name='MicropolisWalkEnv-v0',
      version='0.0.1',
      install_requires=['gym']  # And any other dependencies foo needs
)

setup(name='MicropolisArcadeEnv-v0',
      version='0.0.1',
      install_requires=['gym']  # And any other dependencies foo needs
)
