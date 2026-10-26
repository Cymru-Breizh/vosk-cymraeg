export KALDI_ROOT=`realpath ../..`
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin/:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
# pocolm's scripts call each other by bare name (they carry python3 shebangs),
# so scripts/ has to be on PATH. Set after common_path.sh, which rewrites PATH.
export POCOLM_ROOT=$KALDI_ROOT/tools/pocolm
export PATH=$PATH:$POCOLM_ROOT/scripts

export LC_ALL=C

# For now, don't include any of the optional dependencies of the main
# librispeech recipe
