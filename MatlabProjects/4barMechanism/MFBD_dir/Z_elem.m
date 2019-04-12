function [Z] = Z_elem(mim1,ki,mi,type)
% Z=Z_elem(m1,k,m2,'d') compute the 2 ports model Z(s)
%               of the classical 2 masses + 1 spring system,
% Z=Z_elem(m1,k,m2,'i')  compute its inverse,
% Z=Z_elem(m1,k,m2,'iu')  compute the upper chanel inverse,
% Z=Z_elem(m1,k,m2,'il')  compute the lower chanel inverse,
switch type
    case 'd'
        Z=ss([0 1;-ki/mi 0],[0 0;1/mi -1],[-ki/mi 0;ki 0],[1/mi 0;0 -mim1]);
    case 'i'
        Z=inv(ss([0 1;-ki/mi 0],[0 0;1/mi -1],[-ki/mi 0;ki 0],[1/mi 0;0 -mim1]));
    case 'iu'
        Z=ss([0 1;0 0],[0 0;1 -1],[ki 0;ki 0],[mi 0;0 -mim1]);
    case 'il'
        Z=ss([0 1;-ki/mi-ki/mim1 0],[0 0;1/mi 1/mim1],[-ki/mi 0;ki/mim1 0],[1/mi 0;0 -1/mim1]);
    otherwise
        disp('Wrong syntax !!');Z=[];
end
end

