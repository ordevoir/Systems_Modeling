# Simplest Environment with PyGame

Создадим простейшую среду  с визуализацией через PyGame, используя Parallel API, на основе этого [tutorial](https://pettingzoo.farama.org/tutorials/custom_environment/) (в котором нет PyGame). Так же можно посмотреть [этот образец для создания среды](https://pettingzoo.farama.org/content/environment_creation/).

## Project Structure

Рекомендуемая структура проекта:

```
Simplest-Env
├── simplest-env
    └── env
        └── simplest_env.py
    └── simplest_env_v0.py
├── README.md
└── requirements.txt
```

В `env` сохраняется среда, вместе со всеми вспомогательными функциями. Файл `simplest_environment_v0.py` импортирует среду – имя файла используется для контроля версии. В `requirements.txt` отслеживаются зависимости среды. Как минимум, здесь должен быть `pettingzoo`. Рекомендуется указывать версии зависимостей через `==`.

>готовая к установке среда также должна содержать [дополнительные файлы](https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/#advanced-additional-optional-files)

Вся логика среды должна храниться в директории `env`.

Однако, для данного простого проекта, все реализовано в одном файле `simplest_env.py`, который расположен в корне проекта. Для его тестирования можно запустить сам файл.

Сама среда представляет собой класс, в котором определено поле класса `metaclass` (словарь), и некоторой набор публичных методов:


```python
from pettingzoo import ParallelEnv

class raw_env(ParallelEnv):
    metadata = {
        "name": "simplest_environment_v0",
    }

    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
```

## Environment Logic

Создадим среду с двумя агентами: `prisoner` и `guard`. Первый пытается сбежать в `escape`, а второй должен его перехватить. Среда представляет собой сетку $m \times m$, в которой:

- `prisoner` стартует в левом верхнем углу,
- `guard` стартует в правом нижнем углу,
- `escape` случайным образом располагается в средней части сетки.
- оба агента могут двигаться одном из четырех направлений (`up`, `down`, `left`, `right`).

### `metadata`

Поле класса `metadata` хранит константы среды.

### `__init__()`

Конструктор принимает аргументы среды, к примеру, режим отрисовки `render_mode`. 

### `reset()`

Метод производит сброс к стартовой точке, а в начале игры – производит инициализацию атрибутов так, чтобы при вызове методов `render()`, `step()` и `observe()` не возникало проблем. В нашей среде необходимо инициализировать следующие атрибуты:

    - agents
    - timestamp
    - prisoner x and y coordinates
    - guard x and y coordinates
    - escape x and y coordinates
    - observation
    - infos

Список агентов `agents` копируется из `possible_agents` и в общем случае может меняться в процессе игры, хотя в данной среде такое не предусмотренно.

Метод должен возвращать два объекта `observations, infos`, и даже если информация не предполагается, следует возвращать некоторый фиктивный объект для конвертации `parallel_to_aec`.

### `step()`

В методе производится выбор действия для текущего агента (который определяется классом `agent_selection`). Метод принимает словарь `agent: action`, в котором определены действия для соответствующих агентов.

Метод выполняет выбранные извне действия, т.е. в соответствтии с этими действиями, необходимо обновить среду:

    - prisoner x and y coordinates
    - guard x and y coordinates
    - terminations
    - truncations
    - rewards
    - timestamp
    - infos

а также все внутренние состояния, которые используются методами `observe()` и `render()`.

При этом необходимо проверять некоторые условия:
- первым делом проверяются **termination conditions** для каждого агента; условие завершения агента определяются дизайном игры;
- вторым делом проверяются **truncation conditions** – условие искусственного завершения игры, обычно, когда превышего максимальное число шагов, ограничение по времени, или же исчерпание всех ресурсов. 

Цикл игры продолжается до тех пор, пока в списке `agent` остаются элементы, поэтому для завершения игры, необходимо сделать этот список пустым.

Метод `step()` должен возвращать пять объектов: `observations, rewards, terminations, truncations, infos`.

Мы проводим валидацию внутри `step()`, чтобы недопускать выход агента за пределы сетки. Также мы применяем Action Masking, который описан ниже.

### `render()`

В этом методе производится визуализация среды.

### `observation_space()` & `action_space()`

Метод `observation_space()` должен возвращать пространство наблюдений агента, имя которого передано в качестве аргумента. 

Метод `action_space()` должен возвращать пространство действия агента, имя которого передано в качестве аргумента.

Пространства в gymnasium описанны [здесь](https://gymnasium.farama.org/api/spaces/). Здесь мы использовали классы `Discrete` и `MultiDiscrete` из `gymnasium.spaces`.

Декоратор `lru_cache` из `functools` позволяют произвести мемоизацию пространств наблюдений и действий, сокращая такт циклов, требующих получаения пространств от каждого агента.


## Action Masking

Как вариант, можно просто проверять, прежде чем менять среду, являются ли выбранные действия для агентов валидными, и уже исходя из этого менять или не менять среду. Но более естественный способ – передавать в словаре `observations` маску действий. Это позволит обрабатывать недопустимые действия снаружи.

Для этого, после того, создается массив из единиц, длина которого равна числу всевозможных действий агента. Далее, в зависимости от расположения агента (после совершения текущего действия), необходимо указать действия, которые будут выводить за пределы сетки (а `guard` еще и не должен занимать позицию `escape`). В данной среде 4 всевозможных действия кодируются значениями из множества $\{0, 1, 2, 3\}$. Для недопустимых действий в маскировочном массиве устанавливается значение 0 под соответствующим индексом.

В нашей среде генерируются две маски `prisoner_action_mask` и `guard_action_mask`. Эти массивы размещаются под ключом `"action_mask"` в словарях соответствующих агентов.

## Testing Environment

Для тестирования можно добавить в конце файла следующий код:

```python
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = raw_env()
    parallel_api_test(env, num_cycles=1_000_000)
```

   
## Embed PyGame

В конструкторе класса среды можно задать размер экрана. Там же или в `metadata` можно задать fps. Кроме того, в конструкторе объявляются переменные `screen` и `clock`, и сразу инициализируется `pygame`, если задан режим `"rgb_array"`:

```python
self.render_mode = render_mode
self.screen_size = screen_size
self.screen = None
self.clock = None
```

В рассмтриваемом примере реализованы два режима рендеринга (`"human"` и `"rgb_array"`).

Если задан режим `"human"`, то визуализация производится посредством окна PyGame. Окно запускается при вызове метода `render()`, либо при запуске цикла среды, в вызовах метода `step()`. При этом, после окончания циклов среды окно PyGame не закрывается тут же, а дожидается закрытия окна пользователем.

В режиме `"rgb_array"` вызов метода `render()` вовзращает трехмерный массив NumPy, который может быть визуализирован функцией `imshow()` из `matplotlib.pyplot`.

### `render()`

Рассмотрим фрагмент основного метода `render()`:

```python
def render(self, static=True):
    self.screen_size = (self.screen_size // self.grid_size) * self.grid_size
    if self.render_mode == "human":
        self.human_render(static)
    elif self.render_mode == "rgb_array":
        return self.array_render()
```

В первой инструкии корректируется значение `screen_size`, чтобы оно было кратно `grid_size`. Далее в зависимости от режима, вызывается `human_render()` или `array_render()` в зависимости от выбранного режима.

### `human_render()`

В методе производится установка некоторых переменных, если они еще не установленны, и производится отрисовка (размещение элементов сцены выделено в отдельнуый метод `draw()`):

```python
if self.screen is None:
    self.clock = pygame.time.Clock()
    self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
    pygame.display.set_caption("Simplest Environment")
self.draw()
pygame.display.flip()
```
Далее необходимо обрбатывать событие закрытия окна (тип `pygame.QUIT`). И здесь есть два варианта:
- статическая визуализация, когда производится визуализация неподвижного состояния среды;
- динамическая визуализация, когда производится визуализация эволюционирующей среды.

```python
if static:
    running = True
    while running:
        self.clock.tick(self.metadata["render_fps"])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                self.close()
else:
    self.clock.tick(self.metadata["render_fps"])
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.close(truncate=True)
```

При статической визуализации запускается собственный циклы, в котором производится визуализация и отслеживание события закрытия окна. Закрытие окна при этом не приводит к отсечению (*truncation*) в среде, а лишь сбрасывает переменные PyGame (см. метод `close()` ниже).

При динамической визуализации, PyGame использует в качестве цикла игры цикл среды, и отрисовка запускается на каждом шаге в методе `step()`. В этом случае, закрытие окна приводит также к отсечению в среде.

### `step()`

В методе `step()` среды вызывается `render()` если задан режим `"human"`:

```python
if self.render_mode == "human":
    self.render(static=False)
```
Так как `static=False` в `render()` производится отрисовка и отслеживания события закрытия окна. При завершении игры, сновы запускается `render()`, но тут уже `static=True`:

```python
if any(terminations.values()) or all(truncations.values()):
    self.agents = []
    self.render(static=True)
```

Это необходимо для того, чтобы окно PyGame не закрывалась сразу же после окончания игры, а дожидалась закрытия окна пользователем в собственном цикле.

### `close()`

Метод сбрасывает переменные, используемые для окна PyGame, и всегда вызывается при закрытии окна. При этом, если `truncate=False`, производится завершение игры, путем опустошения списка `self.agents`:
```
def close(self, truncate=False):
    if self.screen is not None:
        pygame.quit()
        self.screen = None
        self.clock = None
        if truncate:
            self.agents = []
```

>Для запуска в различных режимах, используйте файл `test_siemplest_env.ipynb`