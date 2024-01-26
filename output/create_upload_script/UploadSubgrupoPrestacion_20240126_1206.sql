INSERT INTO [Datalake].[Lake].[SubgrupoPrestacion] (IdSubgrupoPrestacion, ClasificacionSubgrupo, UsuarioModificacion, FechaCreacion, FechaModificacion, Vigencia)
VALUES
    (-1, 'SIN REGISTRO', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (1, 'CIRUGIA', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (2, 'DE DIAGNOSTICO', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (3, 'ECOGRAFIA (ULTRASONIDO)', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (4, 'MEDICINA ESPECIALIDAD', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (5, 'MEDICINA GENERAL', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (6, 'MEDICINA NUCLEAR', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (7, 'PLANIGRAFIA', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (8, 'RAYOS X (RADIOGRAFIA)', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (9, 'RESONANCIA MAGNETICA', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (10, 'SALUD MENTAL', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (11, 'TAC (TOMOGRAFIA) (SCANNER)', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (12, 'TERAPEUTICOS', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (13, 'GENERAL', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (14, 'SANGRE', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (15, 'OTRAS MUESTRAS', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (16, 'ORINA', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (17, 'SECRECIONES CORPORALES', @usuariomodificacion, @fechahoy, @fechahoy, 1),
    (18, 'DEPOSICIONES', @usuariomodificacion, @fechahoy, @fechahoy, 1);