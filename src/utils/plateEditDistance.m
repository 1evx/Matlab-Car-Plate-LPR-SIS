function d = plateEditDistance(s1, s2)
    %PLATEEDITDISTANCE Levenshtein distance between two char vectors / strings.

    s1 = char(string(s1));
    s2 = char(string(s2));
    la = numel(s1);
    lb = numel(s2);
    if la == 0
        d = lb;
        return;
    end
    if lb == 0
        d = la;
        return;
    end
    prev = 0:lb;
    for i = 1:la
        cur = zeros(1, lb + 1);
        cur(1) = i;
        c1 = s1(i);
        for j = 1:lb
            cost = double(c1 ~= s2(j));
            cur(j + 1) = min([cur(j) + 1, prev(j + 1) + 1, prev(j) + cost]);
        end
        prev = cur;
    end
    d = prev(end);
end
