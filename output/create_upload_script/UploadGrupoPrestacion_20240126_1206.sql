INSERT INTO [Datalake].[Prestacion].[GrupoPrestacion] (IdGrupoPrestacion, ClasificacionGrupo, UsuarioModificacion, FechaCreacion, FechaModificacion, Vigencia)
VALUES
    (-1, 'SIN REGISTRO', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (1, 'CONSULTAS MEDICAS', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (2, 'IMAGENOLOGIA', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (3, 'PROCEDIMIENTOS', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (4, 'LABORATORIO', @usuariomodificacion, @fechahoy, @fechahoy, 1);
