# ClassDumpAgent — выгрузка расшифрованных классов игры

Агент сохраняет байткод всех не-JDK классов в момент их загрузки JVM.
Кастомная JVM игры (`exbojdk`) расшифровывает `classes.jar` до вызова
`ClassFileTransformer`, поэтому на диск попадают уже расшифрованные `.class`.

## Сборка (уже собран)

`classdump.jar` собран из `ClassDumpAgent.java`. Пересобрать:

```
javac ClassDumpAgent.java
jar cfm classdump.jar MANIFEST.MF ClassDumpAgent.class ClassDumpAgent$1.class
```

## Запуск

PowerShell, перед запуском игры:

```powershell
$env:JAVA_TOOL_OPTIONS = "-javaagent:C:\SC_Map_Dump-main\tools\classdump\classdump.jar=C:\SC_Map_Dump-main\tools\classdump\dump"
```

Затем запустить игру как обычно. JVM подхватит переменную автоматически и
выгрузит классы в `dump\`.

## Предупреждение

Игра онлайн, с анти-читом (VMProtect). Java-агент изменяет загрузку классов
JVM и может быть расценён как вмешательство — есть риск бана аккаунта.
Запускать на свой страх и риск, желательно на тестовом аккаунте.

## Что потом

Выгруженные `.class` декомпилировать (например CFR / Procyon), найти классы
по `type_id` (207447, 53243, 284865, ... из `class_chain` в
`docs/north_mines_geometry_by_block_id.json`), извлечь AABB-таблицы форм
ProBuilder и подставить в `load_shape_tables` в `tools/mdat_obj_export.py`.
