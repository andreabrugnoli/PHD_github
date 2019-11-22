 function renameFiles

path=('/home/a.brugnoli/GitProjects/MatlabProjects/PH/ReductionPHDAE/KP_Experiment/');
files=dir(path)
for id=3:length(files);
 newName=strcat(files(id).name, '.mat');
movefile( fullfile(path, files(id).name), fullfile(path, sprintf(newName)) ); 
end
