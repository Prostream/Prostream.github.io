# TypeScript 最佳实践

## 类型定义
```typescript
interface User {
  name: string;
  age: number;
}

// 使用类型
const user: User = {
  name: "张三",
  age: 25
};
```

## 泛型使用
```typescript
function getFirst<T>(arr: T[]): T {
  return arr[0];
}
``` 