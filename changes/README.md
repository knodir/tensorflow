This folder contains changes made by Nodir et al.

To start working with this repository clone the `dev` branch, as follows:
`git clone -b dev https://github.com/knodir/tensorflow.git`

To make changes, create a branch (e.g., `new-branch`) from the `dev`, make your changes,
and push back to the `dev` branch (not `master`).
```
git checkout -b new-branch dev
# make your changes
git push origin dev
```


How `dev` branch was created?

We fork off version v1.4.0 (the latest release as of now, Jan. 13, 2018) to create the `dev` branch:
`git checkout tags/v1.4.0 -b dev`.
The `master` branch will track the upstream `master` (to pull the latest changes).
Here is how your branches should look like.
```
$ git remote --verbose
origin  https://github.com/knodir/tensorflow.git (fetch)
origin  https://github.com/knodir/tensorflow.git (push)
upstream        git@github.com:tensorflow/tensorflow.git (fetch)
upstream        git@github.com:tensorflow/tensorflow.git (push)
```

Vincent's instrumentation for data send and receive can be found in sendRecvLogging branch. It was done on v1.2.1
version. This branch is protected in Github to prevent accidental merge/deletes.
