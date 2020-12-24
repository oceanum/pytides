rm -r docs/*
pdoc --html  --force --output-dir docs pytides2
mv docs/pytides2/* docs
rmdir docs/pytides2