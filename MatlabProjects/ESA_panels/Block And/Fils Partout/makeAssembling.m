function[Kas,Mas,N] = makeAssembling(nx,ny,Kel,Mel)
% function[Kas,Mas] = makeAssembling(nx,ny,Kel,Mel)
%
% Creation of Connectivity Matrix and Assembling of Stiffness and Mass
% matrices for 4nodes-12dof elements
%
% Example for a 4x5 elements plate: global node convention
%
%  25----26----27----28----29----30
%   | 16 |  17 |  18 |  19 |  20 |
%  19----20----21----22----23----24
%   | 11 |  12 |  13 |  14 |  15 |
%  13----14----15----16----17----18
%   | 6  |  7  |  8  |  9  |  10 |
%   7----8-----9-----10----11----12
%   | 1  |  2  |  3  |  4  |  5  |
%   1----2-----3-----4-----5-----6
%

% Total number of elements
ne = nx*ny;

% Connectivities (global node numbers for element e)
ConneGl = zeros(ne,4);
for e = 1:ne
    % Element Row
    R = floor((e-1)/nx)+1;
    % Element Column
    C = mod((e-1),nx)+1;
    i1 = (R-1)*(nx+1)+C;
    i2 = i1+1;
    i3 = R*(nx+1)+C+1;
    i4 = i3-1;
    ConneGl(e,:) = [i1 i2 i3 i4];
end

% Connectivities (global dofs)
n = 12; % number of dof per element
N = 3*(nx+1)*(ny+1);   % total nuber of dof

C = zeros(ne,n);
for e = 1:ne
     for i = 1:4
        C(e,3*i-2) = 3*(ConneGl(e,i)-1)+1;
        C(e,3*i-1) = 3*(ConneGl(e,i)-1)+2;
        C(e,3*i) = 3*(ConneGl(e,i)-1)+3;
     end
end


% Total Kinetic and Mass Matrices
Kas = zeros(N,N); Mas = zeros(N,N);
for e = 1:ne
    for i = 1:n
        for j = 1:n
            Kas(C(e,i),C(e,j)) = Kas(C(e,i),C(e,j)) + Kel(i,j);
            Mas(C(e,i),C(e,j)) = Mas(C(e,i),C(e,j)) + Mel(i,j);
        end
    end
end

end