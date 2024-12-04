


# # Loop through each device ID and corresponding session
# for i in {1..7}; do
#     tmux send-keys -t "$i" "CUDA_VISIBLE_DEVICES=$i python render.py -m outputs/${paths[$i]}" C-m
# done


# paths=("pinecone" "counter" "teatime" "garden-new" "ramen" "figurines")

path=$1
cuda_device=$2

root=style-ims
for style_im in "$root"/*; do
    CUDA_VISIBLE_DEVICES=$cuda_device python edit_object_style_transfer.py -m outputs/$path \
        --config_file "config/object_style_transfer/$path.json" --skip_test \
        --style_image $style_im
done

