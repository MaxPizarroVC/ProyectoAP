SELECT  dp.CodigoPrestacion,
		gp.IdGrupoPrestacion,
		gp.ClasificacionGrupo,
		sp.IdSubgrupoPrestacion,
		sp.ClasificacionSubgrupo,
		ap.IdAperturaPrestacion,
		ap.ClasificacionApertura
FROM [Datalake].[Lake].[DetallePrestacion] dp
INNER JOIN [Datalake].[Lake].[SubgrupoAperturaPrestacion] sap ON sap.IdSubgrupoApertura = dp.IdSubgrupoApertura
INNER JOIN [Datalake].[Lake].[GrupoSubgrupoPrestacion] gsp ON gsp.IdGrupoSubgrupo = dp.IdGrupoSubgrupo
INNER JOIN [Datalake].[Lake].[GrupoPrestacion] gp ON gp.IdGrupoPrestacion = gsp.IdGrupoPrestacion
INNER JOIN [Datalake].[Lake].[SubgrupoPrestacion] sp ON (sp.IdSubgrupoPrestacion = sap.IdSubgrupoPrestacion) and (sp.IdSubgrupoPrestacion = gsp.IdSubgrupoPrestacion) 
INNER JOIN [Datalake].[Lake].[AperturaPrestacion] ap ON ap.IdAperturaPrestacion = sap.IdAperturaPrestacion 
WHERE ap.ClasificacionApertura = 'SEXOLOGIA'