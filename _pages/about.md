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

I am a fourth year CS undergraduate student at Kyunghee University, working with [Prof. Seungkyu Lee](https://khu.elsevierpure.com/en/persons/seungkyu-lee) in [PerCV](http://cvlab.khu.ac.kr/index.html). My research focuses on computer vision and reinforcement learning. 


## GitHub Repositories

{% if site.data.repositories.github_repos %}
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.html repository=repo %}
  {% endfor %}
</div>
{% endif %}