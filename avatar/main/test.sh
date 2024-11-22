cd ../tools/
python prepare_fit_pose_to_test.py --root_path ../output/model_dump/$1/
cd ../main/
python train.py --subject_id $1 --fit_pose_to_test --continue
python test.py --subject_id $1 --fit_pose_to_test --test_epoch 4
cd ../tools/
python eval_neuman.py --output_path "../output/result/"+$1+"_fit_pose_to_test" --subject_id $1