"""Entry point: process user task and show top-5 API matches."""

from __future__ import annotations

import sys
from typing import List

from text_handling import TaskVectors, process_task
from tool_usage import score_apis


def main() -> None:
	print("Please enter an English task description: ")
	user_input = input().strip()
	if not user_input:
		print("No input provided. Exiting.")
		sys.exit(0)

	result: TaskVectors = process_task(user_input)

	print("Normalized text:", result.normalized_text)
	print("a_T vector:", result.a_t)
	print("z_sem dim:", result.z_sem.shape)

	top_matches = score_apis(result.a_t, result.z_sem, top_k=5)

	print("\nTop-5 APIs by capability match:")
	for rank, (api, score) in enumerate(top_matches, start=1):
		print(f"{rank}. {api.name} (id={api.id}) | score={score:.4f}")


if __name__ == "__main__":
	main()

"""
测试样例：
1️⃣ 时间 / 日期类（应该匹配 GetToday / GetDate）
What is today's date?
Please tell me the current date.
Get today's date for me.
What day is it today?
Tell me the current day and date.

2️⃣ 消息 / 通知类（SendMessage / SendIM）
Send a message to my friend saying I will arrive late.
Please send an instant message to John.
Send a short message to notify the user.
Help me send a message to my colleague.
Send an IM to my team with the meeting update.

3️⃣ 邮件类（SendEmail / ReceiveEmail）
Send an email to Alice about the meeting tomorrow.
Please help me send an email to my manager.
Check if I have received any new emails.
Retrieve my latest emails.
Receive unread emails from my inbox.

4️⃣ 注册 / 账户类（Register / CancelRegistration）
Register a new account for the user.
Help me create a new user registration.
Cancel my current registration.
Unregister my account from the service.
Delete my registration information.

5️⃣ 查询 / 检索类（Search / Query）
Search for user information by name.
Query the database for order details.
Retrieve information about a specific user.
Look up the account record in the system.
Find the details of the last transaction.

6️⃣ 对照组：明确“不应该匹配任何 API”的任务

这些是负样本，你应该看到“乱选”，这是对的：

Write a C++ program to print Hello World.
Explain how attention mechanism works.
Prove the convergence of gradient descent.
Translate this paragraph into Chinese.
Summarize the following academic paper.
"""

