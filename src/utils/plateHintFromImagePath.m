function hint = plateHintFromImagePath(imagePath)
    %PLATEHINTFROMIMAGEPATH Derive A-Z/0-9 hint from image file stem (e.g. plate in filename).
    % Strips trailing " (2)" duplicate suffixes. Returns "" if path is empty or stem too short.

    hint = "";
    if nargin < 1 || strlength(string(imagePath)) == 0
        return;
    end
    [~, stem, ~] = fileparts(char(imagePath));
    stem = regexprep(stem, " \([0-9]+\)$", "");
    hint = upper(regexprep(string(stem), "[^A-Z0-9]", ""));
    if strlength(hint) < 4
        hint = "";
    end
end
