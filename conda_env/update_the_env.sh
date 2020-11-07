# If you had to update or install any packages,
# Run this script so that the changes are pushed with your
# code change and other people can use the new packages
cp conda_env.yml "conda_env.yml.backup_$(date +"%Y-%m-%d-%T")"
conda env export > conda_env.yml 
