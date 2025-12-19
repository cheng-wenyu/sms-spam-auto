# train_model.py - 修正版，兼容Notebook和独立运行
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow

def main(experiment_name="github_actions_demo"):
    """主训练函数 - 使用内置数据快速验证"""
    print(f"🚀 开始自动化训练运行: {experiment_name}")
    
    # 1. 使用sklearn内置数据模拟（保证可运行）
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. 设置MLflow（如果是GitHub Actions，默认记录到本地）
    # mlflow.set_tracking_uri("databricks") # 如果要连回你的Databricks，取消注释并配置Token
    
    with mlflow.start_run(run_name=experiment_name):
        # 3. 训练一个简单模型
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. 评估
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 5. 记录到MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 10)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✅ 训练成功！准确率: {accuracy:.4f}, AUC: {auc:.4f}")
        print(f"   模型已记录到MLflow")
    
    return True

# === 关键修改：判断运行环境，兼容Notebook ===
if __name__ == "__main__":
    # 检查是否在可能包含内核参数的Notebook环境中
    is_likely_notebook = any('ipykernel' in arg or 'json' in arg for arg in sys.argv)
    
    if is_likely_notebook:
        # 在Notebook中直接调用，不使用argparse
        print("检测到Notebook环境，直接运行...")
        success = main()
        sys.exit(0 if success else 1)
    else:
        # 在命令行中运行，使用argparse解析参数
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment-name", type=str, default="auto_run")
        args = parser.parse_args()
        
        try:
            success = main(args.experiment_name)
            sys.exit(0 if success else 1)
        except Exception as e:
            print(f"❌ 失败: {e}")
            sys.exit(1)
