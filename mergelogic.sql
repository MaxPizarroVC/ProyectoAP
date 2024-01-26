MERGE INTO DesarrolloBi.Prestaciones.DesarrolloClasificacionPrestacion AS target
USING (
    VALUES --Resultados del script 
        ('value1', 'value2'),
        ('value3', 'value4'),
        -- Add more rows as needed
) AS Actualizadorquenovive (CodigoVC Varchar(25), GlosaCorregida Varchar(255), GlosaProcesoVC, GlosaImed, ClasificacionGrupo, ClasificacionSubgrupo, ClasificacionApertura, FechaPrestacion, Vigencia, UsuarioModificacion, FechaCreacion, FechaModificacion)
ON (DesarrolloBI.Prestaciones.DesarrolloClasificacionPrestacion.CodigoVC = Actualizadorquenovive.CodigoVC)
AND (DesarrolloBI.Prestaciones.DesarrolloClasificacionPrestacion.Vigente = Actualizadorquenovive.Vigente)
WHEN MATCHED AND DesarrolloClasificacionPrestacion.ClasificacionGrupo != coso subgrupo, lo mismo para coso subgrupo, coso apertura
THEN    UPDATE SET
            DesarrolloBI.Prestaciones.DesarrolloClasificacionPrestacion.Vigente = 0 ,
            DesarrolloBI.Prestaciones.DesarrolloClasificacionPrestacion.FechaModificacion = Actualizadorquenovive.FechaModificacion
WHEN NOT MATCHED THEN
    INSERT (CodigoVC, GlosaCorregida, GlosaProcesoVC, GlosaImed, ClasificacionGrupo, ClasificacionSubgrupo, ClasificacionApertura, FechaPrestacion, Vigencia, UsuarioModificacion, FechaCreacion, FechaModificacion)
    VALUES (resultado del script);