
install

```
% uv venv
Using CPython 3.13.3
Creating virtual environment at: .venv
✔ A virtual environment already exists at `.venv`. Do you want to replace it? · yes
Activate with: source .venv/bin/activate
alexskene@Alexs-TT-MBP aville-model % source .venv/bin/activate
(aville-model) alexskene@Alexs-TT-MBP aville-model % uv sync
Resolved 58 packages in 9ms
Installed 56 packages in 86ms
 + aiohappyeyeballs==2.6.1
 + aiohttp==3.11.18
 + aiosignal==1.3.2
...
% uv lock
Resolved 58 packages in 2ms
```

run

show roi
```
uv run appleville-model.py --roi --prestige-level=2
```

main model is run via
```
uv run appleville-model.py > log.txt
```
or for turn-by-turn simulation
```
uv run appleville-model.py --debug > log.txt
```

add this to get a chart of the runs
```
--chart
```