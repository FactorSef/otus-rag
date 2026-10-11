# Как подключить BLTouch к Raspberry Pi в роли микроконтроллера
### Зачем?
- некоторые платы принтеров не поддерживают управляющий сигнал BLTouch или требуют выпаять какие-то компоненты. В этом случае придётся немного повозиться, но это возможно — так почему бы и нет?

### Что понадобится
- BLTouch
- плоскогубцы или что-то подобное, чтобы извлечь один пин из разъёма Dupont

### Raspberry Pi как MCU
- настройте Raspberry Pi как MCU по [этому руководству](https://www.klipper3d.org/RPi_microcontroller.html)

### Подключение BLTouch
- извлеките красный пин из трёхконтактного разъёма кабеля и вставьте его в одиночный корпус Dupont (есть в оригинальной упаковке BLTouch)
- подключите провода к [выводам GPIO](https://pinout.xyz/):
  - белый -> GPIO0
  - чёрный -> GND рядом с GPIO0
  - жёлтый -> GPIO27
  - коричневый -> GND
  - красный -> 3.3V
  - *TODO: добавить фото проводов*
- для control_pin можно выбрать любой вывод, но проще следовать инструкции, если выбрать те же

### Изменения в printer.cfg
- добавьте следующее в printer.cfg (см. также [справочник по конфигурации](https://www.klipper3d.org/Config_Reference.html#bltouch))
- 
```bash
[mcu rpi]
serial: /tmp/klipper_host_mcu

[bltouch]
sensor_pin: ^rpi:gpio0
#   Pin connected to the BLTouch sensor pin. Most BLTouch devices
#   require a pullup on the sensor pin (prefix the pin name with "^").
#   This parameter must be provided.
control_pin: rpi:gpio27
#   Pin connected to the BLTouch control pin. This parameter must be
#   provided.
z_offset: 0.2
#pin_up_touch_mode_reports_triggered: False
#probe_with_touch_mode: True 

```

### Удачный выбор выводов для датчика
- для sensor_pin можно выбрать gpio0 или gpio1: у них есть встроенная подтяжка (pullup), которая всегда включена
- Klipper по разным причинам не умеет включать подтяжку на GPIO
- пока я этого не знал, здесь было длинное описание, как включать подтяжку скриптом и следить, чтобы этот скрипт всегда запускался.
### Проверка датчика
- следуйте [официальному руководству по проверке BLTouch](https://www.klipper3d.org/BLTouch.html#initial-tests)

### Завершение
- замените endstop_pin в секции [stepper_z] на
```
endstop_pin: probe:z_virtual_endstop
```

- задайте правильное смещение относительно сопла по [официальному руководству](https://www.klipper3d.org/Probe_Calibrate.html#calibrating-probe-z-offset)
