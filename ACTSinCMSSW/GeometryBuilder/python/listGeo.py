#!/usr/bin/env python3
import pkgutil, Configuration.Geometry; print([m.name for m in pkgutil.iter_modules(Configuration.Geometry.__path__) if "Geometry" in m.name and "Extended" in m.name])