# Volunteer Scheduling
Um software de alocação de voluntários a serviço da Vila Residencial (UFRJ)

## Intro
Como medida de contingência à [COVID-19](https://www.worldometers.info/coronavirus/), desde o dia 25 de março o [Restaurante Universitário](https://ru.ufrj.br/) da UFRJ passou a oferecer as refeições a um público mais restrito, na forma de quentinhas entregues ao Hospital Universitário, ao Alojamento Estudantil e à Vila Residencial. Em particular para este último público, 220 a 250 quentinhas têm sido entregues duas vezes ao dia (para almoço e janta) na Associação de Moradores da Vila Residencial, na Cidade Universitária - RJ.

Com isso, estudantes têm se oferecido para ajudar na distribuição das quentinhas que chegam a cada turno, sendo oportuno, portanto, um sistema que permita a alocação de voluntários para dividir nossos esforços de modo a facilitar a participação de todos segundo a disponibilidade de cada um, evitando possíveis sobrecargas e garantindo que os alunos tenham acesso à alimentação provida pela Pró-Reitoria de Gestão e Governança [(PR6 - UFRJ)](https://gestao.ufrj.br/index.php/estrutura-administrativa/quem-somos).

## Organização
A modelagem matemática atual e alguns testes são apresentados em um [jupyter notebook](https://jupyter.org/) disponível no diretório `presentation` do projeto.
O código-fonte da solução do problema está disponível no diretório `src`.

### Dependências
São dependências do código as bibliotecas
- [numpy](https://numpy.org/)
- [cvxpy](https://www.cvxpy.org/)
- [matplotlib](https://matplotlib.org/) _(futuramente)_

Além disso, fazemos uso do solver comercial [Gurobi](https://www.gurobi.com/) (sob licença acadêmica - UFRJ) para solução de problemas de [otimização linear](https://en.wikipedia.org/wiki/Linear_programming) com [variáveis inteiras](https://en.wikipedia.org/wiki/Integer_programming) [(binárias)](https://www.cvxpy.org/tutorial/advanced/index.html?highlight=boolean#mixed-integer-programs).
O solver foi instalado em ambiente Linux [(Xubuntu 19.10)](https://xubuntu.org/about/), conforme [este passo a passo](https://paste.ubuntu.com/p/9fFNzRDQ59/).
