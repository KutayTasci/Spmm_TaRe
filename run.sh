# variables
exec_path="./cmake-build-release/Spmm_TaRe"
folder_path="/home/serdar/Code/PycharmProjects/DM-Partition/folders/" # do not forget the slash at the end
declare -a dsets=("pattern1-7-0.3")
declare -a comm_types=("op" "tp")
declare -a reduces=("noreduce" "reduce")
prc_count=7
# END of variables
# print headers
max_length=0
for dset in "${dsets[@]}"
do
    if [ ${#dset} -gt $max_length ]; then
        max_length=${#dset}
    fi
done

# adjust dname_text
dname_text="dname"
printf -v dname_text "%-${max_length}s" $dname_text
printf -v other_headers "%-4s,%-5s,%-8s,%-8s,%-8s,%-8s,%-8s,%-8s,%-8s,%-8s,%-8s" "comm" "reduce" "min_run" "max_run" "avg_run" "p1_calc" "p1_comm" "p2_calc" "p2_comm" "fin_calc" "total"
echo -e "$dname_text,$other_headers"
# run the program
for dset_name in "${dsets[@]}"
do
    dset=$folder_path$dset_name
    for comm_type in "${comm_types[@]}"
    do
        for reduce in "${reduces[@]}"
        do
          # op does not support reduce
          if [ "$comm_type" == "op" ] && [ "$reduce" == "reduce" ]; then
            continue
          fi
          mpirun -np $prc_count $exec_path $dset $comm_type $reduce 100 50
        done
    done
done