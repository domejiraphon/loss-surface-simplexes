
python base_trainer_svhn.py -model_dir base/lenet/pf0.5 -tensorboard -pf 0.5 -lenet -restart
python simplex_trainer_all.py -model_dir simplex/bad_mode_good_con -load_model base/pf0.5/100.pt -tensorboard -pf 0
python simplex_trainer_all.py -model_dir simplex/lenet/bad_mode_good_con -load_model base/lenet/pf0.5/250.pt -tensorboard \
        -pf 0 -lenet -restart -scale 1e-3 -lr 1e-4
python complex_iterative.py -load_dir base/lenet/pf0.5 -tensorboard -pf 0.5 -lenet -model_dir complex/bad_mode_bad_con -scale 1e-3 -restart
#govind
python complex_iterative.py -model_dir complex/good_mode_good_con -lenet -pf 0 -load_simplex 
python simplex_trainer_all.py -model_dir simplex/lenet/good_mode_good_con -lenet -pf 0 -load_simplex 
#plot
python complex_iterative.py -model_dir complex/good_mode_good_con -lenet -pf 0 -plot

python simplex_trainer_all.py -model_dir simplex/lenet/good_mode_good_con -lenet -pf 0 -plot
run @v16g0 python complex_iterative.py -pf 0 -lenet -load_dir mix_1 -model_dir complex2/1_good_mode_3_bad_mode -restart
run @v17g0 python simplex_trainer_all.py -load_dir trained_model/lenet/pf0.5/0 -tensorboard -lenet -pf 0.1 -model_dir simplex/bad_mode_bad_con_pf0.1 
run @v17g0 python simplex_trainer_all.py -plot -lenet  -model_dir simplex2/lenet/good_mode_good_con -pf 0
run @v17g1 python complex_iterative.py -test -resnet  -model_dir simplex2/resnet/1_good_mode_3_bad_mode -pf 0
run @v16g0 python complex_iterative.py -pf 0 -resnet -load_dir mix_1 -model_dir complex2/resnet/1_good_mode_3_bad_mode -restart
run @v16g0 python complex_iterative.py -pf 0 -vgg -load_dir mix_4 -tensorboard -model_dir complex2/vgg/4_good_mode_0_bad_mode -restart