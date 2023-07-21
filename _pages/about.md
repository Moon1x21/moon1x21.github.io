---
layout: about
title: about
permalink: /


profile:
  align: right
  image: prof_pic.jpg
  image_circular: true # crops the image to make it circular

news: true  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: true  # includes social icons at the bottom of the page
---

I am a first-year M.S. student in [Artifical intelligence](https://ai.postech.ac.kr/main/index) at POSTECH, advised by [Prof. Namhoon Lee](https://namhoonlee.github.io). My research focuses on efficient learning and model optimization in mahicne learning to provide robust and provable solutions to challenging problems, especially in large-scale settings.


## GitHub Repositories

{% if site.data.repositories.github_repos %}
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.html repository=repo %}
  {% endfor %}
</div>
{% endif %}