
python base_trainer_svhn.py -model_dir base/lenet/pf0.5 -tensorboard -pf 0.5 -lenet -restart
python simplex_trainer_all.py -model_dir simplex/bad_mode_good_con -load_model base/pf0.5/100.pt -tensorboard -pf 0
python simplex_trainer_all.py -model_dir simplex/lenet/bad_mode_good_con -load_model base/lenet/pf0.5/250.pt -tensorboard \
        -pf 0 -lenet -restart -scale 1e-3 -lr 1e-4python simplex_trainer_all.py -model_dir simplex/lenet/bad_mode_good_con \
        -load_model base/lenet/pf0.5/250.pt -tensorboard -pf 0 -lenet -restart -scale 1e-3 -lr 1e-4