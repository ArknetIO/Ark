## 📦 Arklang v0.05 — Changelog

### ✨ Added: Generics (Schemas)

* Generic schema definitions using `<T>` / `<A, B>`
* Monomorphization: concrete LLVM structs emitted per used instantiation (lazy on first use)

```ark
schema Box<T> { val: T }

let b1 = Box<i32> { val: 42 };
let b2 = Box<f64> { val: 3.14159 };
```

---

### 📦 Added: Imports / Modules

* File imports with namespace aliasing
* Qualified access via `alias.symbol`
* Modules are **not** runtime values (attempting to use them like values is a compile-time error)

```ark
import "math.ark" as math; // Vector is defined

let v = Vector { x: 1, y: 2 };
math.add(10, 20);
```

---

### 🛠️ Fixed: Value Loading (No More “Pointer Bug”)

* Locals stored in stack slots are now loaded before use, so SSA values flow correctly (e.g., `print x` prints the value, not an address)

```ark
let x = 123;
print x; // ✅ prints 123 (not 0x7ffe...)
```

---

### 🔧 Improved: Unified Member Access

* One member-access path handles:

  * Modules: `math.pi`
  * Enums: `Color.Red`
  * Structs: `p.x`
  * Generics (mangled): `box.val`
  * Builtins: `vec.len`

```ark
print math.pi;
print p.x;
print box.val;
```

---

### 🧠 Parser Updates

* `<...>` is correctly treated as generics in schema defs and instantiations (not “less-than”)
* `FS`, `IO`, `NET` no longer block variable names (capability tokens handled separately)

```ark
schema Pair<A, B> { first: A, second: B }
let p = Pair<i32, f64> { first: 10, second: 2.5 };
```

---

### ⚠️ Breaking

* `struct` is removed; use `schema` only

```ark
schema Point { x: i32, y: i32 }
```
