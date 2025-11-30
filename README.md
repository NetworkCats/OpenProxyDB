# OpenProxyDB

A crawler that collects proxy-related IP blocks from Wikipedia's local and global block lists.

## Overview

OpenProxyDB automatically fetches IP addresses that have been blocked by Wikipedia for proxy-related reasons (VPN, Tor, webhost, etc.) and exports them to a CSV file. The crawler runs daily via GitHub Actions and publishes the results to GitHub Releases.

## Usage

Download the latest `proxy_blocks.csv` from the [Releases](../../releases) page.

## CSV Format

```
ip,anonblock,proxy,vpn,cdn,public-wifi,rangeblock,school-block,tor,webhost
192.168.1.1,False,True,True,False,False,False,False,False,False
10.0.0.0/8,False,False,False,False,False,False,False,True,True
```

## Contributing

This project does not accept any pull requests related to IP classification correction.

## License

CC0-1.0
