source bootstrap.sh
source venv/bin/activate
rm -rf _build
make html
deactivate
