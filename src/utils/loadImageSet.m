function imageSet = loadImageSet(imageRoot, annotationFile)
%LOADIMAGESET Load images and optional annotations for experiments or tests.

arguments
    imageRoot (1,1) string
    annotationFile (1,1) string = ""
end

if ~isfolder(imageRoot)
    error("loadImageSet:MissingFolder", "Image folder does not exist: %s", imageRoot);
end

imageFiles = [ ...
    dir(fullfile(imageRoot, "**", "*.jpg")); ...
    dir(fullfile(imageRoot, "**", "*.jpeg")); ...
    dir(fullfile(imageRoot, "**", "*.png")); ...
    dir(fullfile(imageRoot, "**", "*.bmp"))];

annotations = table();
if strlength(annotationFile) > 0 && isfile(annotationFile)
    annotations = readtable(annotationFile, TextType="string");
end

imageSet = struct( ...
    "path", strings(0,1), ...
    "name", strings(0,1), ...
    "annotation", cell(0,1));

for i = 1:numel(imageFiles)
    filePath = string(fullfile(imageFiles(i).folder, imageFiles(i).name));
    row = table();
    if ~isempty(annotations)
        matchMask = annotations.image_name == string(imageFiles(i).name);
        if any(matchMask)
            row = annotations(find(matchMask, 1, "first"), :);
        end
    end

    imageSet(end+1).path = filePath; %#ok<AGROW>
    imageSet(end).name = string(imageFiles(i).name);
    imageSet(end).annotation = {row};
end
end

