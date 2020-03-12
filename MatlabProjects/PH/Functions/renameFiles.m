 function renameFiles(path)

files=dir(path)
for id=3:length(files);
    name_file = files(id).name;
    if ~contains(name_file, '.')
        newName=strcat(name_file, '.mat');
        movefile( fullfile(path, name_file), fullfile(path, sprintf(newName)) );
    end
end
