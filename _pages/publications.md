---
layout: page
permalink: /publications/
title: publications
nav: true
years: [2020, 2019, 2018, 2017]
---

<div class="publications">

{% for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[hidden_year={{y}}]* %}
{% endfor %}

</div>
