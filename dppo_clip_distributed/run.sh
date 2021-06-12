#!/bin/sh
set -e

cd $(dirname $0)

kungfu_flags() {
    echo -q
    echo -logdir logs

    local ip1=127.0.0.1
    local np1=$np

    local ip2=127.0.0.10
    local np2=$np
    local H=$ip1:$np1,$ip2:$np2
    local m=cpu,gpu

    echo -H $ip1:$np1
}

prun() {
    local np=$1
    shift
    kungfu-run $(kungfu_flags) -np $np $@
}

n_leaner=2
n_actor=2
n_server=1

flags() {
    echo -l $n_leaner
    echo -a $n_actor
    echo -s $n_server
}

rl_run() {
    local n=$((n_leaner + n_actor + n_server))
    prun $n python3 start2.py $(flags)
}

main() {
    rl_run
}

main
