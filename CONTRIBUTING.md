# Guía de colaboración

Para evitar que varias personas modifiquen directamente la branch principal, cada integrante debe trabajar en su propia branch.

## Flujo recomendado

1. Actualizar la branch principal:

   ```bash
   git switch main
   git pull origin main
   ```

2. Crear una branch para la tarea:

   ```bash
   git switch -c nombre/tarea
   ```

   Ejemplos: `nico/modelo-ml`, `ana/eda` o `juan/conclusiones`.

3. Guardar cambios y subirlos:

   ```bash
   git add .
   git commit -m "Describe brevemente el cambio"
   git push -u origin nombre/tarea
   ```

4. Crear un Pull Request en GitHub hacia `main`.
5. Revisar los cambios entre todos y hacer el merge.
6. Antes de comenzar otra tarea, volver a actualizar `main`.

## Trabajo con el notebook

- Coordinar qué sección o celdas modifica cada persona.
- Evitar editar simultáneamente las mismas celdas: los conflictos de archivos `.ipynb` son difíciles de resolver manualmente.
- Hacer commits pequeños y descriptivos.
- Antes de subir cambios desde Colab, verificar que se seleccionó la branch correcta.
- No subir claves, tokens, contraseñas ni información privada.

