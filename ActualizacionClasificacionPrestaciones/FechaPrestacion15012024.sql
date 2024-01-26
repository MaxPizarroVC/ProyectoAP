select CodigoVC, MAX(FechaCreacion) FechaPrestacion
from DesarrolloBI.Prestaciones.PrestacionesVidacamara
group by CodigoVC