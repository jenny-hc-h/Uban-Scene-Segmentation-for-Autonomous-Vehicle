import argparse
import glob 
import os

"""
Check Dataset path with train/val folder
	python checkDataset.py --dataset DATASET_DIR
"""
def main(data_mode_dir):
	im_fpath_train = glob.glob(data_mode_dir+"/train/leftImg8bit/*.png")
	im_fpath_valid = glob.glob(data_mode_dir+"/val/leftImg8bit/*.png")

	num_train = 0
	for i in im_fpath_train :
		lb_fn_train = os.path.splitext(i.split('/')[-1])[0][0:-12] + '_gtCoarse_color.mat'
		lab_fpath_train  = data_mode_dir+"/train/gtCoarse/"+lb_fn_train
		if not os.path.exists(lab_fpath_train):
			print("path not found")
			print(lab_fpath_train)
			break
		num_train = num_train+1
	print("training sample : %d" % num_train)

	num_val = 0
	for i in im_fpath_valid:
		lb_fn_val = os.path.splitext(i.split('/')[-1])[0][0:-12] + '_gtCoarse_color.mat'
		lab_fpath_val  = data_mode_dir+"/val/gtCoarse/"+lb_fn_val
		if not os.path.exists(lab_fpath_val):
			print("path not found")
			print(lab_fpath_val)
			break
		num_val = num_val+1
	print("validate sample : %d" % num_val)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Scene Segmentation')
	parser.add_argument('--dataset',type=str,required=True,help='Specify the directory of dataset')
	args = parser.parse_args()
	main(args.dataset)