# FreestyleParser - диагностика Windows
# Запуск: powershell -ExecutionPolicy Bypass -File win-diagnostic.ps1
# Результат: файл freestyle-info.txt на рабочем столе
# Работает на PowerShell 3.0+ (Windows 7 SP1+), без прав администратора.
# Каждая секция в try/catch: сломаться не может даже наполовину.

$ErrorActionPreference = 'SilentlyContinue'
$info = New-Object System.Collections.Generic.List[string]

function Add-Sec($title) { $info.Add(''); $info.Add('=' * 4 + ' ' + $title + ' ' + '=' * 4) }
function Try-Cmd($title, $script) {
    Add-Sec $title
    try { $out = & $script; if ($out) { foreach ($l in $out) { $info.Add([string]$l) } } else { $info.Add('(пусто)') } }
    catch { $info.Add('ОШИБКА СБОРА: ' + $_.Exception.Message) }
}

$desktop = [Environment]::GetFolderPath('Desktop')
if (-not $desktop) { $desktop = $env:USERPROFILE + '\Desktop' }
$outFile = Join-Path $desktop 'freestyle-info.txt'

Try-Cmd 'ЗАПУСК' { "Диагностика FreestyleParser, $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" }

Try-Cmd 'СИСТЕМА' {
    $os = Get-CimInstance Win32_OperatingSystem
    "Windows: $($os.Caption) $($os.Version) ($($os.OSArchitecture))"
    "Компьютер: $($env:COMPUTERNAME), пользователь: $($env:USERNAME)"
    "PowerShell: $($PSVersionTable.PSVersion)"
    "Свободно на C:: $([math]::Round((Get-PSDrive C).Free/1GB,1)) ГБ"
    "RAM всего: $([math]::Round($os.TotalVisibleMemorySize/1MB,1)) ГБ"
}

Try-Cmd 'ПУТЬ ПОЛЬЗОВАТЕЛЯ (кириллица?)' {
    "USERPROFILE: $env:USERPROFILE"
    $cp = [System.Text.Encoding]::Default.EncodingName
    "Кодировка системы: $cp"
    "Локаль: $(Get-WinSystemLocale)"
}

Try-Cmd 'УСТАНОВЛЕННАЯ ПРОГРАММА (Program Files)' {
    foreach ($p in @("$env:ProgramFiles\FreestyleParser", "${env:ProgramFiles(x86)}\FreestyleParser")) {
        if (Test-Path $p) {
            "Найдено: $p"
            Get-ChildItem $p | ForEach-Object { "  $($_.Name)  $([math]::Round($_.Length/1MB,1)) МБ  $($_.LastWriteTime)" }
        }
    }
}

Try-Cmd 'КОПИИ ПРОГРАММЫ (загрузки/рабочий стол)' {
    Get-ChildItem "$env:USERPROFILE\Downloads\FreestyleParser*",
                  "$env:USERPROFILE\Desktop\FreestyleParser*",
                  "$desktop\FreestyleParser*" -ErrorAction SilentlyContinue |
        ForEach-Object { "$($_.FullName)  ($($_.LastWriteTime))" }
}

Try-Cmd 'ЗАПУЩЕНЫ ЛИ КОПИИ ПРОГРАММЫ СЕЙЧАС' {
    Get-Process | Where-Object { $_.Name -like '*FreestyleParser*' } |
        ForEach-Object { "$($_.Name) PID=$($_.Id) старт=$($_.StartTime)" }
    '(если строк нет - программа не запущена)'
}

Try-Cmd 'НАСТРОЙКИ (~\.FreestyleParser)' {
    $cfg = Join-Path $env:USERPROFILE '.FreestyleParser'
    if (Test-Path $cfg) {
        Get-ChildItem $cfg -Recurse | ForEach-Object {
            if ($_.PSIsContainer) { "  [папка] $($_.Name)" }
            else { "  $($_.FullName.Replace($cfg,'~'))  $([math]::Round($_.Length/1KB,1)) КБ  $($_.LastWriteTime)" }
        }
    } else { '(папки нет - программа ни разу не запускалась или другая учётка)' }
}

Try-Cmd 'ЛОГ ПРИЛОЖЕНИЯ app.log (последние 80 строк)' {
    $log = Join-Path $env:USERPROFILE '.FreestyleParser\app.log'
    if (Test-Path $log) { Get-Content $log -Tail 80 -Encoding UTF8 }
    else { '(нет)' }
}

Try-Cmd 'ПАПКИ ДНЕЙ (Desktop\FreestyleParser)' {
    $base = Join-Path $desktop 'FreestyleParser'
    if (Test-Path $base) {
        Get-ChildItem $base -Directory | ForEach-Object {
            $n = ($_.GetFiles() | Measure-Object -Property Length -Sum)
            "  $($_.Name): $($n.Count) файлов, $([math]::Round($n.Sum/1MB,1)) МБ"
        }
    } else { '(папки дней нет)' }
}

Try-Cmd 'ПОСЛЕДНЯЯ ПАПКА ДНЯ - СОДЕРЖИМОЕ' {
    $base = Join-Path $desktop 'FreestyleParser'
    $last = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($last) {
        "Папка: $($last.FullName)"
        Get-ChildItem $last.FullName | ForEach-Object {
            "  $($_.Name)  $([math]::Round($_.Length/1KB,0)) КБ  $($_.LastWriteTime)"
        }
    } else { '(нет)' }
}

Try-Cmd 'НАСТРОЙКИ ДНЯ processing.yaml' {
    $base = Join-Path $desktop 'FreestyleParser'
    $last = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($last -and (Test-Path (Join-Path $last.FullName 'processing.yaml'))) {
        Get-Content (Join-Path $last.FullName 'processing.yaml') -Encoding UTF8
    } else { '(нет)' }
}

Try-Cmd 'ЛОГ ДНЯ processing.log (последние 60 строк)' {
    $base = Join-Path $desktop 'FreestyleParser'
    $last = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    $pl = Join-Path $last.FullName 'processing.log'
    if ($last -and (Test-Path $pl)) { Get-Content $pl -Tail 60 -Encoding UTF8 }
    else { '(нет)' }
}

Try-Cmd 'FFMPEG В СИСТЕМЕ' {
    $ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ff) { "ffmpeg: $($ff.Source)"; & ffmpeg -version 2>$null | Select-Object -First 1 }
    else { 'ffmpeg не найден в PATH (для программы это нормально - он в комплекте)' }
}

Try-Cmd 'АНТИВИРУС (Defender)' {
    $mp = Get-MpComputerStatus -ErrorAction SilentlyContinue
    if ($mp) {
        "Defender: реал-тайм=$($mp.RealTimeProtectionEnabled), версия сигнатур=$($mp.AntivirusSignatureVersion)"
        "Исключений для нас нет? Это не обязательно, но ускоряет запуск"
    } else { 'Defender недоступен (сторонний антивирус или версия ОС)' }
    Get-Process | Where-Object { $_.Name -match 'avp|avast|avgui|mcshield|kaspersky|360|drweb' } |
        ForEach-Object { "Сторонний антивирус: $($_.Name)" }
}

Try-Cmd 'КОНЕЦ' { "Файл готов: $outFile - перешлите его разработчику" }

# BOM, чтобы блокнот точно показал кириллицу
[System.IO.File]::WriteAllLines($outFile, $info, (New-Object System.Text.UTF8Encoding $true))
Write-Host "`nГотово! На рабочем столе файл freestyle-info.txt - перешлите его нам." -ForegroundColor Green
try { Start-Process notepad $outFile } catch {}
