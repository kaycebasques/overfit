source bootstrap.sh
source venv/bin/activate
[ -d _build ] && rm -rf _build
make html
deactivate
