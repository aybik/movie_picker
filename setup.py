from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='moviepicker',
      version="0.0.1",
      description="Movie Recommendation System",
      license="MIT",
      author="Movie Picker Team",
      author_email="aybike.alkn@gmail.com",
      url="https://github.com/aybik/movie_picker",
      install_requires=requirements,
      packages=find_packages(),
    #   test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
