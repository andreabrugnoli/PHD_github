% Met la figure courante au format A4 et en mode paysage

% Copyright (c) 1993-2000 ONERA/DCSD, All Rights Reserved.
%#

  Ox =  1.0 ;
  Oy =  2.0 ; 
  Px = 29.7 ; 
  Py = 21.0 ;
  Lx=Px-2*Ox;
  Ly=Py-2*Oy;

%  
%   configuration impression
%

  PaperOrientation = 'landscape' ;
  PaperUnits = 'centimeters';
  PaperPosition = [ Ox Oy Lx Ly ] ;
  PaperType = 'a4letter';

%
%   configuration figure
%

 % Position = [ Ox+5 Oy Lx Ly ] ;            % Pour Linux, Unix
  Position = [ Ox-1 Oy-1 0.75*Lx 0.75*Ly ] ; % Pour Windows
  Units=' centimeters';

sauve=get(gcf,'Units');

set(gcf,        'PaperType', PaperType, ...
		'PaperOrientation', PaperOrientation, ...
		'PaperUnits',PaperUnits, ...
                'PaperPosition', PaperPosition, ...
                'Units', 'centimeters', ...
		'Position',Position, ...
                'Units', sauve ...
)  ;





