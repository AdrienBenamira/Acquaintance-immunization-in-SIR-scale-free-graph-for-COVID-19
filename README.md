# Acquaintance-immunization-in-SIR-scale-free-graph-for-COVID-19

## Slides + Demo

Please look at the slides for more infos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/AdrienBenamira/Acquaintance-immunization-in-SIR-scale-free-graph-for-COVID-19/blob/master/demo_online.ipynb)

## Worst case scenario: temproal results

![img1](https://github.com/AdrienBenamira/Acquaintance-immunization-in-SIR-scale-free-graph-for-COVID-19/blob/master/img/Epidemic%20curve%20over%20time%20if%20nothing%20happens.png)

#### Comments : 

1- If nothing is done, according this modelisation (imperfect of course), we will finish  with 69% of the population removed. That’s perfectly match with the worst case scenario proposed by the [NYT](https://www.nytimes.com/2020/03/13/us/coronavirus-deaths-estimate.html): 224 millions Americans can be infected by the virus vs 226 millions with this estimation.

2- With no exterior agent, the epidemic should stop a time 18, which represent 60% of the population removed. That's means, if 60% of our population is removed, the endemic state is over.



## Answer to the question: If a vaccine against CO-VID is found tomorrow, which vaccination strategy leads to the suppression of the endemic state for a lowest immunization rate ?

Comparaison strategies

![img1](https://github.com/AdrienBenamira/Acquaintance-immunization-in-SIR-scale-free-graph-for-COVID-19/blob/master/img/plot%20final%20comparaison%20strategy.png)


|                                                                                    | Random | Targeted |  Acquitance K = 20% |
|:----------------------------------------------------------------------------------:|:------:|:--------:|:-------------------:|
| Percentage of the population  vaccinacte in order to stop the endemic state |  85%`  |    5%    |         30%         |


#### Comments : 

1- Random immunisation is not an efficient strategy

2- Targeted the hubs of the networks are super efficient strategy, but implies that we know the graph (which is not true)

3- Acquitance strategy gives good results and this strategy is purely local, requiring minimal information about randomly selected nodes and their immediate environment.

## References

```
@article{cohen2003structural,
  title={Structural properties of scale free networks},
  author={Cohen, Reuven and Havlin, Shlomo and Ben-Avraham, Daniel},
  journal={Handbook of graphs and networks},
  volume={4},
  publisher={Wiley Online Library}
}

@article{cohen2000resilience,
  title={Resilience of the internet to random breakdowns},
  author={Cohen, Reuven and Erez, Keren and Ben-Avraham, Daniel and Havlin, Shlomo},
  journal={Physical review letters},
  volume={85},
  number={21},
  pages={4626},
  year={2000},
  publisher={APS}
}

@article{cohen2003efficient,
  title={Efficient immunization strategies for computer networks and populations},
  author={Cohen, Reuven and Havlin, Shlomo and Ben-Avraham, Daniel},
  journal={Physical review letters},
  volume={91},
  number={24},
  pages={247901},
  year={2003},
  publisher={APS}
}

@article{madar2004immunization,
  title={Immunization and epidemic dynamics in complex networks},
  author={Madar, Nilly and Kalisky, Tomer and Cohen, Reuven and Ben-avraham, Daniel and Havlin, Shlomo},
  journal={The European Physical Journal B},
  volume={38},
  number={2},
  pages={269--276},
  year={2004},
  publisher={Springer}
}

```

[Simulate an empidemie  - 3Blue1Brown](https://www.youtube.com/watch?v=gxAaO2rsdIs&t=549s)


[Github repo](https://github.com/maufadel/SIR_on_Gnutella)


