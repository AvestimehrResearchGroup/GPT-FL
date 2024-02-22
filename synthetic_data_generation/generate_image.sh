export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

for device_num in 6; do
    CUDA_VISIBLE_DEVICES=0, taskset -c 1-60 python generate_image.py --gpu_number 10 --device_num $device_num --generate_method ucg --dataset food_101
done
# CUDA_VISIBLE_DEVICES=1, taskset -c 1-60 python generate_image.py --gpu_number 10 --device_num 1 --generate_method class_prompt multi_domain ucg --dataset food_101
# CUDA_VISIBLE_DEVICES=0, taskset -c 0-30 python generate_image.py --gpu_number 10 --device_num 5