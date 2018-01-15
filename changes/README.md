This folder contains changes made by Nodir et al.

We fork off version v1.4.0 (the latest release as of now, Jan. 13, 2018) to create the `dev` branch:
`git checkout tags/v1.4.0 -b dev`. All of our changes will be done on the `dev` branch.
The `master` branch will track the upstream `master` (to pull the latest changes).
Here is how your branches should look like.
```
$ git remote --verbose
origin  https://github.com/knodir/tensorflow.git (fetch)
origin  https://github.com/knodir/tensorflow.git (push)
upstream        git@github.com:tensorflow/tensorflow.git (fetch)
upstream        git@github.com:tensorflow/tensorflow.git (push)
```

To make changes, create a branch (e.g., `new-branch`) from the `dev`, make you changes,
and push back to the `dev` branch (not `master`).
```
git checkout -b new-branch dev
# make your changes
git push origin dev
```

Vincent's instrumentation for data send and receive can be found in sendRecvLogging branch. It was done on v1.2.1
version. This branch is protected in Github to prevent accidental merge/deletes.
