docs: Home of documentation for Resource Superstaq
===================================================
This repository contains materials for Resource Superstaq.

## How to build the docs locally
### Setup your environment

Clone the repository and set up your virtual environment

    git clone git@github.com:Infleqtion/resource-superstaq.git
    cd resource-superstaq
    python3 -m venv venv_resource_superstaq
    source venv_resource_superstaq/bin/activate
    python3 -m pip install -e .     
    python -m pip install -r docs/requirements.txt

### Build the docs
1.  From the parent `resource-superstaq` directory. Run `checks/build_docs.py`
0. Run `open build/html/index.html`
 
## How to update the docs
1. Make sure you are on the `main` branch in `resource-superstaq`.
0. Create a new branch off of `main` in which to make your updates.
0. Make any relevant updates.
0. Push all commits and create a Pull Request.
0. Request the relevant people to review your Pull Request.
0. After your Pull Request has been reviewed, merge in your branch.